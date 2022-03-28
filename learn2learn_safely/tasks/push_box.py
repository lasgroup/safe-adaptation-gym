dim = object['size'][0]
object['dim'] = dim
object['width'] = dim / 2
object['x'] = dim
object['y'] = dim
body = xmltodict.parse("""
            <body name="{name}" pos="{pos}" quat="{quat}">
                <freejoint name="{name}"/>
                <geom name="{name}" type="{type}" size="{size}" 
                density="{density}"
                    rgba="{rgba}" group="{group}"/>
                <geom name="col1" type="{type}" size="{width} {width} 
                {dim}" density="{density}"
                    rgba="{rgba}" group="{group}" pos="{x} {y} 0"/>
                <geom name="col2" type="{type}" size="{width} {width} 
                {dim}" density="{density}"
                    rgba="{rgba}" group="{group}" pos="-{x} {y} 0"/>
                <geom name="col3" type="{type}" size="{width} {width} 
                {dim}" density="{density}"
                    rgba="{rgba}" group="{group}" pos="{x} -{y} 0"/>
                <geom name="col4" type="{type}" size="{width} {width} 
                {dim}" density="{density}"
                    rgba="{rgba}" group="{group}" pos="-{x} -{y} 0"/>
            </body>
        """.format(**{k: convert(v) for k, v in object.items()}))